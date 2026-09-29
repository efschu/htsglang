# [Deprecated] Re-export shim for backward compatibility. Use dsa.triton_kernel instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.triton_kernel is deprecated; "
    "use flliper.srt.layers.attention.dsa.triton_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.triton_kernel import *  # noqa: F401, F403
