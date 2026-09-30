# [Deprecated] attention/nsa/ is a thin re-export shim for backward compatibility.
# Use attention/dsa/ instead. This directory will be removed in a future release.
import warnings

warnings.warn(
    "flliper.srt.layers.attention.nsa is deprecated; "
    "use flliper.srt.layers.attention.dsa instead.",
    DeprecationWarning,
    stacklevel=2,
)
from flliper.srt.layers.attention.dsa import *  # noqa: F401, F403
