"""Unified public kernel namespace for fLLiper (RFC #29630).

fLLiper runtime code and tests should import callable kernels from
``flliper.kernels.ops.<group>``, e.g.::

    from flliper.kernels.ops.layernorm import rmsnorm
    from flliper.kernels.ops.activation import silu_and_mul
    from flliper.kernels.ops.kvcache import reshape_and_cache_flash

Implementations still live in ``flliper.jit_kernel`` (JIT CUDA), the
``sgl_kernel`` wheel (AOT CUDA/C++), Triton op modules, etc. The ``ops.*``
functions are thin wrappers that forward to a chosen backend; the
:data:`~flliper.kernels.registry.registry` provides an inventory of every
backend so alternatives can be compared and selected. Multi-backend operators
are structured as :class:`~flliper.kernels.fused_op.BaseFusedOp` subclasses —
one ``forward_<backend>`` method per backend behind a single ``forward()``,
with a required pure-``torch`` ``forward_native`` reference and a
``FLLIPER_FORCE_FUSED_OP_BACKEND`` global switch.

Importing this package (and any ``ops.*`` group) does not import a kernel
backend (``sgl_kernel`` / ``flliper.jit_kernel``) or trigger JIT compilation:
registration is metadata-only and backends are imported lazily on first call.
This keeps the namespace usable for inventory tooling on a CPU-only box.
"""

from flliper.kernels.fused_op import (
    BaseFusedOp,
    FusedOpTraceRecord,
    clear_fused_op_trace,
    disable_fused_op_trace,
    enable_fused_op_trace,
    get_fused_op_backend,
    get_fused_op_trace,
    register_fused_op,
    set_fused_op_backend,
)
from flliper.kernels.registry import KernelRegistry, register_kernel, registry
from flliper.kernels.selector import get_kernel, select_kernel
from flliper.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
)

# Importing the operator groups populates the registry (metadata only). Kept
# after the core imports above (and guarded from isort reordering) so those
# modules are fully initialized first.
from flliper.kernels import ops  # noqa: E402  # isort: skip

__all__ = [
    "ops",
    "BaseFusedOp",
    "CapabilityRequirement",
    "FormatSignature",
    "FusedOpTraceRecord",
    "KernelBackend",
    "KernelRegistry",
    "KernelSpec",
    "PlatformInfo",
    "clear_fused_op_trace",
    "disable_fused_op_trace",
    "enable_fused_op_trace",
    "get_fused_op_backend",
    "get_fused_op_trace",
    "get_kernel",
    "register_fused_op",
    "register_kernel",
    "registry",
    "select_kernel",
]
