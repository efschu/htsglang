"""``sglang.kernels.jit.utils`` -> this line's ``sglang.jit_kernel.utils``."""

from sglang.jit_kernel import utils as _utils
from sglang.jit_kernel.utils import (  # noqa: F401  (the names the ops import)
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


def __getattr__(name):
    return getattr(_utils, name)
