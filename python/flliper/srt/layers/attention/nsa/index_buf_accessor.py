# [Deprecated] Re-export shim for backward compatibility. Use dsa.index_buf_accessor instead.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa.index_buf_accessor is deprecated; "
    "use flliper.srt.layers.attention.dsa.index_buf_accessor instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa.index_buf_accessor import *  # noqa: F401, F403
