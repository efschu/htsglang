# [Deprecated] Re-export shim for backward compatibility. Use dsa.tilelang_kernel instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.tilelang_kernel is deprecated; "
    "use flliper.srt.layers.attention.dsa.tilelang_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.tilelang_kernel import *  # noqa: F401, F403
